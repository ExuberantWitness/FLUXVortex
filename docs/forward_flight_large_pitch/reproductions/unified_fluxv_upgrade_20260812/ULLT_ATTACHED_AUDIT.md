# One-state ULLT attached-flow audit

## Status

The source-constrained one-state model is implemented and independently
executable.  It does not read digitized loads, does not accept a paper/case
identifier, and uses only the constants published by Izraelevitz et al.:

- `A_phi=-0.5`, `b_phi=-0.25`;
- `A_Gamma=-0.8`, `b_Gamma=-0.25`;
- lifting-surface correction `K=13.5`; and
- flat-plate added-mass correction `K_am=0.85`.

The implementation includes tilted semi-infinite horseshoe trailers, the
one-state lift/circulation dynamics, vector lift/induced drag, equation-(42)
surface correction, and equations-(35)--(39) added mass.  Circulatory and
added-mass forces are exported separately.

## Izraelevitz Figure 11 no-fit result

The model was run at 256 steps/cycle for four cycles, with the last cycle
evaluated against the separately digitized curves only after prediction.

| comparison | lift metric | drag metric |
|---|---:|---:|
| paper UVLM, raw phase RMSE | 0.1546 `CLalpha` | 0.3142 `CDalpha` |
| paper UVLM, range-NRMSE | 1.36% | 8.82% |
| digitized paper 1-state, raw phase RMSE | 0.0766 | 0.0295 |

Predicted ranges are `CLalpha=-5.839..5.839` and
`CDalpha=-3.658..-0.224`; the digitized paper UVLM ranges are approximately
`CLalpha=-5.670..5.685` and `CDalpha=-3.485..0.071`.

This closes the reduced model strongly enough to use it as an independently
executable numerical reconstruction for Figure 11.  It is not an independent
experimental validation and remains an inviscid attached-flow model, not a
separated-flow model.

## Yang 2025 transfer and shared-gate audit

The exact same implementation was run at all six Yang angles of attack.
No parameter was changed by angle, paper, or geometry.  The exploratory 15--20
degree local incidence gate was then used to examine a transparent load-level
blend with UVLM plus the nonlinear polar residual.

| model | lift MAE (gf) | drag MAE (gf) |
|---|---:|---:|
| one-state ULLT alone | 8.339 | 12.774 |
| UVLM + nonlinear polar | 3.951 | 2.063 |
| ULLT/UVLM-polar load-level blend | 3.897 | 2.783 |

The blend slightly improves lift MAE but materially worsens drag MAE.  It does
not pass as the unified production upgrade.  The reason is structural: a
global exported Ptera history does not expose separate attached-circulatory
and non-circulatory pressure channels.  Blending total loads removes part of
the UVLM history and added-mass ledger along with the desired attached load.

Recommended use:

1. retain one-state ULLT as an independently executable numerical Figure 11 baseline;
2. retain UVLM plus nonlinear polar as the current unified candidate for Yang;
3. only revisit ULLT as a FLUXV replacement after implementing a panel/strip
   circulatory-pressure decomposition inside the solver, so UVLM wake memory
   and non-circulatory loads remain exactly preserved.

## Reproduction

Core model:

`platform/forward_flight_benchmarks/ullt_attached.py`

Audit runner:

```bash
PYTHONPATH=src:platform python -m \
  forward_flight_benchmarks.run_ullt_attached_audit \
  --output docs/forward_flight_large_pitch/reproductions/\
unified_fluxv_upgrade_20260812/ullt_attached_audit.json
```

Tests:

```bash
cd platform
PYTHONPATH=../src python -m pytest \
  tests/test_ullt_attached.py \
  tests/test_uvlm_polar_correction.py \
  tests/test_forward_flight_benchmarks.py -q
```

The final unified regression selection reports `23 passed` after adding
phase-boundary and appended-cycle invariance tests.
