# FluxV v4b findings

## Evidence status

This file records the frozen result-to-claim interpretation.  Numerical values
are populated from the final v4b manifests after deterministic hash checks and
same-family independent recomputation (provisional). Earlier `v4`, `v4_v2`,
`v4_v3` and variable-core runs are failed development evidence and are not
headline results.

Fresh-agent gates: `result-to-claim=partial` (high confidence) and
`integrity=WARN/qualified_with_warnings`; both are same-family provisional
reviews. The deterministic evidence precheck found all six cited values.

## Supported findings

1. The author LDVM v2.5 executable/reference case can be rebuilt and reproduces
   the bundled force output to approximately `1e-7`; the clean-room Python
   primitive reaches the correct first LESP cap but remains partial parity
   (`172` versus `174` LEVs).
2. Both audited dissertations are two-dimensional.  They provide useful LEV
   section physics, not a published 3-D low-order validation case.
3. A standalone 2-D LDVM replacement is not adequate for a finite wing:
   Stevens' leading-edge-axis smoke case improved while its mid-chord case
   degraded.  Retaining UVLM and adding only a paired separation discrepancy
   is the supported architecture.
4. The source-resolved v4b implementation fixes the v4a audit blockers:
   phase-level ownership, component force ledger, Yang force projection,
   fixed vortex core and Figure-11 identity.

## Frozen numerical table

| Evidence | Metric | Old FluxV | v4b | Interpretation |
|---|---:|---:|---:|---|
| Yang 2025 wind tunnel | lift MAE [gf] | 6.8549 | 4.5545 | improved, but worse than v1 |
| Yang 2025 wind tunnel | drag MAE [gf] | 12.9217 | 2.6440 | large improvement, but worse than v1 |
| Scherer / Figure 14 experiment | mean thrust RMSE | 0.05115 | 0.02595 | 49.3% development-set improvement |
| Izraelevitz Figure 11 numerical reference | lift RMSE | 2.4310 | 0.1546 | attached ULLT limit, not LDVM evidence |
| Izraelevitz Figure 11 numerical reference | drag RMSE | 0.9531 | 0.2293 | attached ULLT limit, not experiment |
| Stevens Figure 21 LE-axis experiment | lift RMSE | 1.0540 | 0.8592 | 18.5% development-transfer improvement |
| Stevens Figure 21 mid-chord experiment | lift RMSE | 0.5472 | 0.4394 | 19.7% development-transfer improvement |

Every measured channel is reported separately. No pooled score, significance
test over correlated curve pixels or unmeasured Stevens drag accuracy is
claimed. Figure 14 and Yang were used during development. Stevens was inspected
after the first smoke run, so it is a development transfer set rather than an
untouched confirmation set.

## Claim boundary

Even if all development gates pass, the strongest supported claim is:

> A phase-resolved, source-assisted LDVM discrepancy improves the retained
> FluxV UVLM benchmark envelope on the reconstructed development cases and
> improves Stevens lift transfer, subject to the reported discretization and
> data limitations.

The results do not establish a conservative 3-D material-LEV UVLM coupling,
universal LESP parameter, independent confirmation, numerical convergence or
production readiness. The next scientific step is a genuinely held-out finite-
wing case with experimental phase-resolved lift and drag, followed by a shared
3-D LEV/wake state rather than a stripwise additive discrepancy.
