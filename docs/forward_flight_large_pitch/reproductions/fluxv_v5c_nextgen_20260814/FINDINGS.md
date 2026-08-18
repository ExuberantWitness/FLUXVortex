# FluxV v5c research findings

## v5c0: source-ledger correction

Figure 14 uses the published `0.75c` reference for both the polar incidence and
the single `Cd0=0.057` ledger. Correcting the legacy `0.25c` reference changed
the 14-marker CT RMSE from `0.025949167` to `0.024250786` without using the
observations for parameter selection. This is a correctness repair, not a new
separation model.

## v5c1: rate-sensitive axial-suction loss

The source-frozen mechanical model passed exact-off, attached-flow, hot-state
decay, causality, boundedness, axial-suction sign, periodic-state and refinement
tests. The final pole is `0.5` under `Delta tau=(U/c)dt`; the causal rate is a
first-order backward difference. These results support the implementation
contract only.

The frozen 22-condition cache proxy failed the preregistered cross-paper
non-regression gate:

Canonical negative artifact:
`runs/20260814_fluxv_v5c1_proxy_all22_pole05_rate1_reproducible/`. Its 21
declared source/dependency hashes and six result hashes match, and the exported
previous-cycle state reproduces the reported periodic-state residual.

| Primary metric | Baseline | v5c1 proxy | Outcome |
|---|---:|---:|---|
| Yang lift MAE, gf | 4.554509817 | 4.555307528 | worse |
| Yang drag MAE, gf | 2.643997474 | 2.629697899 | better |
| Figure-14 all-marker CT RMSE | 0.024250786 | 0.024298130 | worse |
| Baik filtered macro CL RMSE | 0.657541867 | 0.657511938 | marginally better |
| Baik filtered macro CD RMSE | 0.345152419 | 0.345147094 | marginally better |

Figure-14 theta-15, theta-25 and unique-condition RMSE rows also worsened, and
Baik W2-CD/W4-CD failed their channel gates. The correction is therefore not a
three-paper improvement. Its input is also noncanonical: it lacks the
same-time-layer UVLM-induced strip velocity and complete `0.75c` local velocity.

## Supported and unsupported claims

Supported:

- the v5c0 `0.75c`/single-`Cd0` ledger correction is internally consistent;
- the v5c1 mechanical state and axial-suction ledger pass their unit/mechanical
  contracts;
- in the explicitly noncanonical proxy, Yang drag improves slightly and Baik
  macro CL/CD changes are slightly favorable.

Not supported:

- v5c1 simultaneously improves Yang 2025, Figure 14 and Baik W1--W4;
- v5c1 is a canonical FluxV result or is eligible for production promotion;
- the tiny mixed changes demonstrate generalization or physical superiority.

## Postmortem and constraints

The published SVDVM relation motivates an LE circulation-flux magnitude after
onset, but it does not supply a validated subcritical fast-rotation trigger.
The v5c1 supra-critical gate consequently leaves the important Figure-14
high-rate/low-`A0` residual nearly untouched, while its small suction reduction
slightly harms other conditions.

Do not retune the pole, threshold or rate scale on the same 22 conditions. A
future candidate must either close a literature-sourced missing mechanism or
repair exclusive attached/LEV/persistent load ownership, with all parameters
frozen before scoring and with same-time-layer strip fields available.

Result-to-claim trace:
`.aris/traces/result-to-claim/2026-08-14_run03/`.

## v5e: ULLT-to-UVLM line-item replacement

The isolated v5e implementation removed Ptera's complete `F_dGamma` panel
term and inserted the source-derived one-state phi--Gamma mismatch plus one
provenance-tagged kinematic added-mass term.  Its disabled replay, panel/strip/
airplane ledgers, force identity and periodic state all passed; 42 targeted
tests and every declared source/result hash passed independent review.

Representative accuracy nevertheless triggered the preregistered stop.  At
Yang AoA 15 degrees lift improved slightly but drag worsened.  At Figure 14
theta=15 degrees, psi=60 degrees, the corrected-baseline CT error increased
from `0.012047` to `0.142447`.  Baik W2 filtered CL RMSE increased from
`1.032306` to `1.542894`, although CD improved.  These regressions are much
larger than the corresponding smoke/full numerical differences, so the
candidate was not run over all 22 conditions and was not tuned.

The precise result and component decomposition are recorded in
`V5E_MECHANICAL_AND_EARLY_STOP_RESULT.md`.  The panel/strip ledger remains a
useful diagnostic, but the force closure itself is archived.  The next route
is a native material-LEV state in the same UVLM AIC/wake/pressure ledger, with
no-LEV bitwise reduction required before paper scoring.
