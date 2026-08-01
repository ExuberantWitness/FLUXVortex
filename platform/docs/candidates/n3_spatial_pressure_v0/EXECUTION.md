# N3 spatial-pressure v0 Fig. 17/18/19 execution

Date: 2026-07-29

This is the executable hand-off for the isolated candidate campaign. It does
not rerun or mutate V4.1. Existing V4.1 artifacts are read only for the
optional visual overlay and post-run delta.

## 1. Frozen inputs and outputs

- Python:
  `/home/exuber/anaconda3/envs/fluxvortex/bin/python`
- working directory:
  `/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV`
- raw measurements:
  `platform/docs/data.md`
- frozen measurement SHA-256:
  `ca4274e0b5c4af4f8fa526f2403579ee5eda1a1cdceabfac5682286f8c3cf3a1`
- read-only V4.1 visual baseline:
  `platform/docs/s6_sweep_v41_full184_20260729_105013.json`
- read-only V4.1 confirmed scorecard:
  `platform/docs/scorecards/scorecard_v41_full184_scoped_20260729_125402.json`
- candidate run root:
  `platform/docs/candidates/n3_spatial_pressure_v0/runs/<timestamp>/`

Every run directory contains `config.json`, `status.json`, and
`candidate_results.json`. The configuration freezes the resolved model/grid,
condition keys, Git state, and implementation source hashes. A source change
therefore cannot silently mix two candidate implementations in one sweep.

## 2. Quick visual route

The following nested route uses one quick-grid identity. `--seed-run` copies
only finite overlapping records after exact model/grid/source identity
validation.

```bash
cd /home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV
PY=/home/exuber/anaconda3/envs/fluxvortex/bin/python

S0=$(date +%Y%m%d_%H%M%S)
R0=platform/docs/candidates/n3_spatial_pressure_v0/runs/$S0
$PY platform/lb_sweep_candidate.py \
  --candidate-id n3_spatial_pressure_v0 \
  --closure n3_spatial_pressure_v0 \
  --scope smoke3 \
  --quick \
  --timestamp "$S0"

S1=$(date +%Y%m%d_%H%M%S)
R1=platform/docs/candidates/n3_spatial_pressure_v0/runs/$S1
$PY platform/lb_sweep_candidate.py \
  --candidate-id n3_spatial_pressure_v0 \
  --closure n3_spatial_pressure_v0 \
  --scope representative32 \
  --quick \
  --seed-run "$R0" \
  --timestamp "$S1"
```

`smoke3` is plumbing only. `representative32` is the first useful visual
package: it contains one complete curve in every Fig. 17/18/19 panel, while
preserving the explicit conditional status of Fig. 19(c,d).

If the 32-point physics and runtime gates pass:

```bash
S2=$(date +%Y%m%d_%H%M%S)
R2=platform/docs/candidates/n3_spatial_pressure_v0/runs/$S2
$PY platform/lb_sweep_candidate.py \
  --candidate-id n3_spatial_pressure_v0 \
  --closure n3_spatial_pressure_v0 \
  --scope confirmed151 \
  --quick \
  --seed-run "$R1" \
  --timestamp "$S2"
```

This produces all 42 source-confirmed curves: all of Fig. 17, all of Fig. 18,
and Fig. 19(a,b). To draw all 50 curves under the explicitly conditional
2.6-Hz interpretation of Fig. 19(c,d), add only the 33 conditional-only
conditions:

```bash
S3=$(date +%Y%m%d_%H%M%S)
R3=platform/docs/candidates/n3_spatial_pressure_v0/runs/$S3
$PY platform/lb_sweep_candidate.py \
  --candidate-id n3_spatial_pressure_v0 \
  --closure n3_spatial_pressure_v0 \
  --scope conditional184 \
  --quick \
  --seed-run "$R2" \
  --timestamp "$S3"
```

The 184 package is visual/diagnostic only. It cannot promote a claim until the
fixed frequency of Fig. 19(c,d) is resolved from authoritative metadata.

## 3. Scoring and plotting

For any run directory `RUN`, write a scope-aware scorecard:

```bash
RUN=platform/docs/candidates/n3_spatial_pressure_v0/runs/<timestamp>
$PY platform/fig171819_benchmark.py \
  "$RUN/candidate_results.json" \
  --data platform/docs/data.md \
  --output "$RUN/fig171819_scorecard.json"
```

Exit code `2` is expected: it means a declared promotion blocker exists
(incomplete global scope and/or unresolved Fig. 19(c,d)), not that scoring
failed. Any other nonzero code is an actual post-processing failure.

Generate measured–V4.1–candidate overlays without overwriting an earlier
image:

```bash
$PY platform/plot_candidate_overlay.py \
  "$RUN/candidate_results.json" \
  --data-md platform/docs/data.md \
  --candidate-label "N3 spatial P2 v0 (quick)" \
  --baseline-json platform/docs/s6_sweep_v41_full184_20260729_105013.json \
  --baseline-label "V4.1 frozen"
```

Outputs are:

- `fig17_candidate_overlay*.png`
- `fig18_candidate_overlay*.png`
- `fig19_candidate_overlay*.png`
- `candidate_overlay_manifest*.json`

The main numerical comparison is the confirmed-scope
`aggregates.ALL.ALL` entry versus the frozen V4.1 confirmed values:
MAE `1.348216637 N`, RMSE `1.578714035 N`, bias `+0.858516583 N`, trend
capture `0.571428571`.

## 4. Formal-grid route

Do not seed a formal run from a quick run; the runner rejects that grid
identity mismatch. After the quick physics gates pass, start a formal
`representative32` run without `--quick`. A later formal `confirmed151` may
seed that formal 32-point run, and a formal `conditional184` may seed the
formal confirmed run.

```bash
$PY platform/lb_sweep_candidate.py \
  --candidate-id n3_spatial_pressure_v0 \
  --closure n3_spatial_pressure_v0 \
  --scope representative32 \
  --timestamp <YYYYMMDD_HHMMSS>
```

The registered production grid is `nc=12`, `ns=16`, four cycles,
`spc_of(U,f,nc)`, and `wake_rows=steps_per_cycle`.

## 5. Runtime budget

Existing V4.1 production evidence gives only a lower-bound reference:

- 66-condition campaign: `7105 s`, or `108 s/case`;
- interrupted fresh campaign: `13745 s / 87 completed`, or `158 s/case`
  including session overhead.

That implies roughly `0.96–1.40 h` for the old closure's production 32,
`4.5–6.6 h` for 151, and `5.5–8.1 h` for 184. The new CPU-side P2 history can
be materially slower and is not credibly estimable before `smoke3`; its cost
grows with sheet history, so a linear V4.1 estimate must not be presented as
the candidate ETA.

For the quick nested route, let `q` be the measured median
`wall_seconds/case` after smoke. The remaining work is:

- representative32: at most `29 q`;
- confirmed151: at most another `119 q`;
- conditional184: at most another `33 q`.

The production campaign is launched only if the sentinel runtime and
quadrature/time-refinement gates make that budget bounded.

## 6. Recovery

- Normal `Ctrl-C` or `SIGTERM`: `status.json` becomes `interrupted`; rerun the
  exact command with `--resume <run-directory>` instead of `--timestamp`.
- `SIGKILL`, host failure, or CUDA reset: `status.json` may remain `running`,
  but every fully returned condition is atomically checkpointed. The same
  `--resume` command skips valid records and retries failures.
- `completed_with_failures`: the process exits nonzero. Resume the same run;
  valid conditions are skipped and failed conditions are retried.
- `aborted_source_drift`: do not resume. The solver/candidate/YAML identity
  changed during the campaign, so a new timestamp is required.
- `--force` intentionally overwrites already valid records and is not part of
  normal recovery.
- `--max-conditions` is debug-only and now records
  `incomplete_debug_prefix`, never `complete`.

No command above writes the frozen V4.1 result, scorecard, or production
figures.
