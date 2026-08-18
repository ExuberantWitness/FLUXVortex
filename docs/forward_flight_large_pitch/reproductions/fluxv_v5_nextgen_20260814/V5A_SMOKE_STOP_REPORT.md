# FluxV v5a smoke stop report

## Verdict

The frozen cache-reuse v5a smoke run is complete.  It passes the pure ledger
closure tests and improves the Yang aggregate means, but fails the predeclared
Figure-14 and Baik gates by large margins.  Per the experiment-plan stop rule,
v5a is frozen as a failed development hypothesis: no v5a full run and no
observation-driven parameter tuning were performed.

This is an auxiliary `kinematic_proxy + projected_integrated_proxy` run.  It
is not a canonical strip-local v5a result because the frozen phase caches do
not contain UVLM-induced strip velocities.

## Frozen smoke result

| Benchmark | v4b | v5a cache smoke | Gate outcome |
|---|---:|---:|---|
| Yang lift MAE [gf] | 4.555 | 3.951 | pass numeric |
| Yang drag MAE [gf] | 2.644 | 2.063 | pass numeric |
| Figure 14, 14-marker CT RMSE | 0.02595 | 0.09470 | fail |
| Baik filtered macro CL RMSE | 0.65754 | 0.80165 | fail |
| Baik filtered macro CD RMSE | 0.34515 | 0.40441 | fail |

Only 6 of 18 numeric/audit gates pass; canonical promotion is 0 of 18.  All
eight Baik case/channel filtered scores regress.  The maximum Baik quadrant
relative regression is 82.35%.  Figure 14's 15/25-degree subgroup RMSE is
0.11297/0.04620 versus gates 0.02268/0.02284.

The result falsifies the current v5a assumption that adding the complete
equilibrium residual and only high-passing the integrated LDVM discrepancy is
sufficient across the three papers.  It does not falsify the unexecuted
canonical strip-local normal/suction formulation, but that formulation cannot
be scored from the frozen caches.

## Reproduction

From the repository's `platform` directory:

```bash
NUMBA_CACHE_DIR=/tmp/numba-v5a \
MPLCONFIGDIR=/tmp/mpl-v5a \
/home/exuber/anaconda3/envs/fluxvortex/bin/python \
  -m forward_flight_benchmarks.run_fluxv_v5a_crosspaper \
  --quality smoke \
  --output-dir ../docs/forward_flight_large_pitch/reproductions/\
fluxv_v5_nextgen_20260814/runs/20260814_fluxv_v5a_cache_smoke_frozen
```

Fast runner contracts:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
NUMBA_CACHE_DIR=/tmp/numba-v5a \
MPLCONFIGDIR=/tmp/mpl-v5a \
/home/exuber/anaconda3/envs/fluxvortex/bin/python \
  -m pytest tests/test_fluxv_v5a_crosspaper.py -q
```

Result directory:

`docs/forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814/runs/20260814_fluxv_v5a_cache_smoke_frozen`

## Result hashes

| File | SHA256 |
|---|---|
| `case_metrics.csv` | `e18df43a46420d4bcba1a9b08d360fa1daae1cdc1dcd208672ad214463c2aaba` |
| `condition_predictions.csv` | `4bdd42be1c21e9bec6857846a9a4623c7aa392f44222ddca85dbfd18181c2e68` |
| `gate_results.csv` | `260fc0e236de7ff862dad4942036bfb07b389276f2ff0bfb6ec03f3e1e89c32b` |
| `phase_histories.csv` | `0cfb3917b318a443046c7c4928a14357542a4ec0ef5b625af2bf13dc256db229` |
| `summary.json` | `4f49811be9649a8bb3a63754353fd54afe2db53f766dfa14a9dd2fcce3d98d15` |
