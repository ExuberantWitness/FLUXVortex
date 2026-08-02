# N2.6-QSSEP quasi-steady separated pressure drag experiment plan

## 1. Objective

- run id: `n26_qssep_v0_shadow_20260801`
- selected idea: add the missing quasi-steady separated pressure drag
  (cross-flow form drag) to N2.6.  Fingerprint (phase3_lesion_T2): T bias
  +0.49/+1.57/+2.24 @ U6/8/10, f-flat, aoa15 covered by N3.3 -> the model
  is missing separated pressure drag that grows with free-stream q=0.5*rho*U^2.
- mechanism (phase3_lesion_T2 ②): DeLaurier 1993 cross-flow drag
  (C_d,cf=1.98 flat plate); Pomerenk & Ristroph 2025 C_D^S=C_D^(pi/2)*sin^2(alpha),
  C_D^(pi/2)~1.8.  f-independent by construction (free-stream q).
- non-negotiable constraints:
  - no retuning to Fig. 17/18/19 (C_D,sep = literature constant 1.8);
  - shadow/minimal: `closure='v41'` untouched bit-exact;
  - separated drag along freestream -x (pure drag; panel-normal leaked +1.94N
    lift -> corrected after G1/G4 probe); gated by loss_frac (N2.1 separated
    fraction) so zero when attached (no N1 double-count);
  - do not revive falsified f^2 sep_drag (candidate C death);
  - preserve frozen V4.1 files.
- research question: can a minimally executable quasi-steady separated
  pressure drag (one literature constant, f-flat, separation-gated) reduce
  the confirmed-scope Fig. 17/18/19 thrust bias without regressing lift?
- null hypothesis: no confirmed-scope metric improves vs V4.1, or any L
  curve degrades >0.15N.
- alternative hypothesis: T MAE/bias improve on the confirmed scope while
  L stays within 0.15N and dT/df shape is preserved.

## 2. Baseline And Comparability

- baseline id: frozen fixed-name `platform/docs/s6_sweep_v41_full184.json`
  (published 2026-08-01, dual-scope contract, promotion_eligible=True).
- baseline variant: `closure=v41`.
- dataset: raw digitized samples parsed from `platform/docs/data.md`.
- scoring: `fig171819_benchmark.scorecard` confirmed scope (42 curves/151
  conditions) vs the fixed-name V4.1 scorecard.
