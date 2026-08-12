# Experiment integrity audit

- date: 2026-08-13
- auditor: fresh GPT-5.6-Sol ultra agent, read-only
- review independence: `same-family`
- acceptance status: `provisional`
- overall verdict: `WARN`
- integrity status: `qualified_with_warnings`

No fabricated values, non-finite results, declared-hash mismatches, or headline
arithmetic errors were found. The auditor independently recomputed every main
metric and ran the specified tests (`23 passed`). The warning is about evidence
scope and model semantics, not phantom results.

## A. Ground-truth provenance: WARN

- Yang, Scherer Figure 14, and Stevens Figure 21 targets are digitized real
  experiments with explicit source identities and frozen CSV hashes.
- Figure 11 is an authors' numerical UVLM reference, not experiment.
- The Stevens/Ramesh/thesis PDFs are not vendored in this publication branch;
  their bibliographic paths, hashes, and digitization methods are documented,
  so PDF-to-CSV reconstruction remains external to a fresh clone.

## B. Metric and weighting integrity: WARN

MAE/RMSE/bias use physical units without prediction-dependent normalization,
and channels are not pooled. Figure 14's primary score preserves all 14 plotted
markers, so two repeated conditions receive double weight. The 14-marker RMSE
changes `0.0511466 -> 0.0259492` (49.3%); after averaging repetitions and giving
12 unique conditions equal weight it changes `0.0435122 -> 0.0275082` (36.8%).
Both are reported. Stevens' 501 samples are correlated curve pixels, not 501
independent experimental repeats.

## C. Result existence, numbers, and hashes: WARN

Independent recomputation reproduced:

- Yang lift/drag MAE: `6.85487/12.92164 -> 4.55451/2.64400 gf`;
- Figure 14 mean-thrust RMSE: `0.0511466 -> 0.0259492`;
- Figure 11 inherited numerical-reference RMSE: `2.43099/0.95313 ->
  0.154624/0.229315`;
- Stevens lift RMSE: `1.05404 -> 0.85918` (LE axis), `0.54718 -> 0.43941`
  (mid-chord axis).

All hashes declared in the 256-step, 512-step, and Stevens-full result metadata
match. `FINAL_PROVENANCE.md` supplements direct dependencies and publication
figures. Final phase-level owner/force histories are not frozen, so those need a
rerun for inspection; only their means are stored.

## D. Dead/stale paths: WARN

- One coefficient helper is test-only.
- A Stevens time-step core-ratio argument is shadowed by the explicitly fixed
  `r_core/c=0.02`; the executed result is correctly fixed-core, but the unused
  argument is misleading.
- Figure 11 imports frozen v2 metric rows rather than producing a new end-to-end
  v4 phase history.

These do not alter the reported numeric results.

## E. Scope and claim fit: WARN

- Phase-level blending is genuinely performed before cycle averaging.
- The paired separated-minus-attached formulation exactly reduces to the
  baseline before LESP onset, but it is still an additive discrepancy with
  independent 2-D wakes. Post-onset common-wake Kelvin conservation and an
  exclusive no-double-count proof are not implemented.
- The Figure-14 threshold is a source-motivated development hypothesis, not an
  independent preregistered validation. Stevens was reclassified from held-out
  to development transfer after the first smoke result.
- Yang aggregate error improves, but lift error worsens at four of six angles
  and v4b does not beat v1 or the authors' model.
- The 256-to-512 fixed-core aggregate check is stable, but it is not a full
  grid/wake/core convergence study; Stevens has no same-code refinement series.

## F. Evaluation classification: PASS

| Evaluation | Classification |
|---|---|
| Yang 2025 wind-tunnel means | `real_gt` |
| Scherer Figure 14 mean thrust | `real_gt` |
| Stevens Figure 21 lift traces | `real_gt`, correlated digitized curves |
| Figure 11 | `numerical_reference`, not experiment |
| Figure-14 local phase cache / model drag | `development_proxy` |
| Ramesh v2.5 author output | `numerical_reference` |
| clean-room LDVM | `numerical_reference_partial_parity` |

## Claim impact

The defensible claim is an aggregate improvement over frozen old FluxV on the
reconstructed development envelope. The evidence does not support universal
generalization, statistical significance, all-condition improvement,
superiority to strong existing variants, or conservative 3-D LDVM–UVLM
coupling. See `CLAIMS_FROM_RESULTS.md` for the frozen wording.
