# N2.6-QSSEP Fig. 17/18/19 execution

Date: 2026-08-01

Executable hand-off for the isolated candidate campaign. Does not rerun or
mutate V4.1. Existing V4.1 artifacts are read only for post-run delta.

## 1. Frozen inputs and outputs

- Python: `/home/exuber/anaconda3/envs/fluxvortex/bin/python`
- working directory: `/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV`
- raw measurements: `platform/docs/data.md` (SHA-256
  `ca4274e0b5c4af4f8fa526f2403579ee5eda1a1cdceabfac5682286f8c3cf3a1`)
- read-only V4.1 baseline: `platform/docs/s6_sweep_v41_full184.json`
- candidate closure: `n26_qssep_v0_shadow` (C_D,sep=1.8, loss_frac-gated,
  freestream -x pure drag)
- outputs: `platform/docs/candidates/n26_qssep_v0_shadow/runs/<ts>/`

## 2. Campaign

```
python platform/lb_sweep_candidate.py \
  --candidate-id n26_qssep_v0_shadow \
  --closure n26_qssep_v0_shadow \
  --scope representative32          # -> confirmed151 -> conditional184
```

## 3. Post-run comparison (single gate, user ruling 2026-08-01)

- score candidate 184 vs fixed-name V4.1 with `fig171819_benchmark.scorecard`
  confirmed scope;
- promotion: as many confirmed-scope metrics improve as possible (T MAE/bias/
  RMSE/trend, L MAE/bias/trend), no L curve degrades >0.15N, dT/df shape
  preserved;
- falsification: no confirmed metric improves and any L degrades >0.15N.
