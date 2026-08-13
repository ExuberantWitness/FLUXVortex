# Baik W1--W4 usage

Run from the repository root with the `fluxvortex` environment:

```bash
PYTHONPATH=src:platform \
MPLCONFIGDIR=/tmp/mpl_baik \
NUMBA_CACHE_DIR=/tmp/numba_baik \
python \
  -m forward_flight_benchmarks.run_baik2012_benchmark \
  --quality full \
  --output-dir \
  docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible
```

The default matrix is W1--W4.  A bounded subset can be selected without
changing the source contract:

```bash
... -m forward_flight_benchmarks.run_baik2012_benchmark \
  --quality smoke --cases W2 W4 --output-dir /tmp/baik_w2_w4_smoke
```

The primary curves and metrics are the model histories passed through the
same sharp 1 Hz Fourier cutoff reported for the experiment.  Raw model
histories are also saved for diagnostics.  The runner never phase-shifts,
rescales or offsets a prediction.

Run the W2 one-factor numerical study with:

```bash
PYTHONPATH=src:platform \
MPLCONFIGDIR=/tmp/mpl_baik \
NUMBA_CACHE_DIR=/tmp/numba_baik \
python \
  -m forward_flight_benchmarks.run_baik2012_sensitivity \
  --output-dir \
  docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/sensitivity/20260813_w2_one_factor_reproducible
```

Run the controlled LDVM time/wake study against the canonical full run with:

```bash
PYTHONPATH=src:platform \
NUMBA_CACHE_DIR=/tmp/numba_baik \
python \
  -m forward_flight_benchmarks.run_baik2012_ldvm_sensitivity \
  --base-run \
  docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible \
  --output-dir \
  docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/sensitivity/20260813_w2_ldvm_controlled_reproducible
```

Runners refuse a non-empty output directory by default.  Use a new directory
for a fresh run; `--allow-existing-output` is an explicit overwrite opt-in.

Targeted regression tests:

```bash
PYTHONPATH=platform \
MPLCONFIGDIR=/tmp/mpl_baik \
NUMBA_CACHE_DIR=/tmp/numba_baik \
python \
  -m pytest -q platform/tests/test_baik2012.py
```

Important identities:

- experiment and model `CD < 0` denote thrust;
- `FluxV old` is the current prescribed-wake UVLM load channel;
- `FluxV v4b` retains that UVLM and adds the declared LDVM/persistent-polar
  transfer;
- `Lcrit=0.11` is the primary Ramesh flat-plate body-text transfer;
- `Lcrit=0.19` is only the conflicting Table 4.1 sensitivity;
- neither threshold is fitted or validated specifically for the Baik plate;
- the Ptera wing is a free-tip mean-surface surrogate for a quasi-2D
  wall/endplate experiment and does not resolve 6.25% thickness or viscosity.
