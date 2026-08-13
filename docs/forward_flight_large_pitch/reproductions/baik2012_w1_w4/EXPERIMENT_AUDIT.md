# Baik 2012 W1--W4 experiment audit

**Date:** 2026-08-13

**Auditor:** fresh same-family Codex agent, read-only

**Acceptance status:** provisional
**Overall verdict:** **WARN / qualified only**

The numerical artifacts and their metric arithmetic pass the integrity audit.
The warning is scientific rather than a result-file failure: this is a
four-condition development-transfer test with an unresolved wall/endplate
boundary, an unvalidated cross-Re LESP threshold, and material wake sensitivity.

## A. Ground-truth provenance: PASS

- Ground truth is digitized from Baik dissertation Figures 5.24--5.27,
  corrected-total direct-force histories, not generated from any model.
- The frozen paired CSV contains 401 points per case; scoring removes only the
  duplicate phase-one endpoint, leaving 400 equally weighted samples per case.
- The extraction retains source JPEGs, source-pixel traces, an overlay, fixed
  calibration code and source/audit hashes.  No missing pixels are completed
  from FluxV or Theodorsen.
- Frozen GT SHA-256:
  `4de6b01cd8072959e5b780053f311efa92ab5a94f17940dd122df340ad638f2f`.

## B. Score normalization and fitting: PASS

- Scores are raw coefficient MAE/RMSE/bias and range-NRMSE; no metric is divided
  by a statistic of the model prediction.
- The model is periodically interpolated to the published phase grid and is
  passed through the source-declared 1 Hz Fourier cutoff.  There is no phase,
  amplitude, scale or offset fit.
- Raw, filtered and pointwise values are all retained.  The duplicate cycle
  endpoint is not double-weighted.

## C. Result existence and arithmetic: PASS

- Canonical full output contains 52 metric rows and 20,800 scored samples.
- A read-only fresh-agent recomputation recovered every reported metric field
  from the pointwise CSV with zero discrepancy.
- All primary filtered and raw old-to-v4b comparisons are finite.  In the
  primary filtered comparison, all four cases improve in both CL and CD RMSE.
- Canonical full manifest: 17/17 declared source hashes and 7/7 result hashes
  match.  Controlled LDVM manifest: 17/17 source/base hashes and 1/1 result
  hash match.  UVLM one-factor manifest: 15/15 and 1/1 match.

Canonical full headline:

| model | macro CL RMSE | macro CD RMSE |
|---|---:|---:|
| FluxV old | 0.694840 | 0.407277 |
| FluxV v4b | 0.657542 | 0.345152 |

This is a 5.37% CL and 15.25% CD RMSE reduction at the frozen settings.

## D. Dead-code and execution-path audit: PASS with minor warning

- The production runner calls the frozen kinematics, old UVPM/UVLM path,
  declared v4b transfer, source-matched filter, metric functions and plotting
  functions; their outputs appear in the canonical CSV/PNG/PDF artifacts.
- Regression tests exercise the case table, W3 typo resolution, nonlinear
  plunge, quarter-chord pivot, effective incidence, filter and GT hash.
- The Ptera wing object name still contains the historical text
  `end-plated rectangular adapter`; the executable and manifest correctly
  identify it as a **free-tip surrogate**.  This naming defect does not change
  geometry or forces but must not be read as a wall/endplate implementation.

## E. Scope and numerical evidence: WARN

- W1--W4 are four correlated conditions from one quasi-2D apparatus, inspected
  during development.  This is not held-out or broad generalization evidence.
- The experiment is a 6.25%-thick rounded plate constrained by walls/endplate;
  FluxV uses a zero-thickness, physical-span free-tip UVLM surrogate.
- `Lcrit=0.11` is a published flat-plate transfer from a different Reynolds
  number/thickness, not a Baik-calibrated value.  `0.19` remains a source-conflict
  sensitivity and is not selected from Baik accuracy.
- The controlled 256/512/1024 LDVM time test at fixed 0.5-cycle wake and fixed
  `rc/c=0.02` changes W2 CL/CD RMSE only about 0.0055/0.0041 from reference to
  1024 steps.  However, changing retained material wake from 0.50 to 0.75 cycle
  changes CL/CD RMSE by about 0.0624/0.0254.  Wake retention is not converged.
- The UVLM 64-to-128 time change is material, and no upward temporal or chordwise
  convergence sequence has been completed.  `full` is a frozen production
  configuration, not a converged solution.

## F. Evaluation classification

- Experimental comparison: `real_gt` via digitized direct-force observations.
- FluxV/Theodorsen histories: numerical-model predictions compared to that GT.
- Theodorsen is a lift-only external reference; it is not experimental data.

## Claim impact

- **Supported:** W1--W4 geometry, nonlinear motion and corrected-total CL/CD
  histories have been reconstructed with an auditable no-fit contract.
- **Supported with qualification:** at the frozen free-tip-surrogate settings,
  v4b improves both CL and CD RMSE in every W1--W4 condition relative to old.
- **Unsupported:** “high absolute accuracy”, “wall/endplate physics reproduced”,
  “Baik confirms a universal LESP threshold”, “held-out generalization”, or
  “mesh/time/wake convergence”.

The canonical evidence directories are those ending in `_reproducible`; older
directories are retained only as deprecated development history.
